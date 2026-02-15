using System;
using System.Collections.Generic;

public static class ObjectResolver
{
    private static readonly Dictionary<string, string> Aliases = new(StringComparer.OrdinalIgnoreCase)
    {
        { "couch", "sofa" },
        { "sectional", "sofa" },
        { "settee", "sofa" },
        { "armchair", "chair" },
    };

    public static string Normalize(string s)
    {
        if (string.IsNullOrWhiteSpace(s)) return "";
        s = s.Trim().ToLowerInvariant();
        if (Aliases.TryGetValue(s, out var mapped)) return mapped;
        return s;
    }

  
    public static SceneObject ResolveTarget(SceneMemory mem, string movedSpan, string category, string targetQuery = null)
    {
        if (mem == null) return null;

 
        var span = Normalize(movedSpan);
        if (!string.IsNullOrEmpty(span))
        {
            var bySpan = mem.FindBestMatchByNameOrSpan(span);
            if (bySpan != null) return bySpan;
        }

       
        var cat = Normalize(category);
        if (!string.IsNullOrEmpty(cat))
        {
            var byCat = mem.GetMostRecentByCategory(cat);
            if (byCat != null) return byCat;
        }

        
        var q = Normalize(targetQuery);
        if (!string.IsNullOrEmpty(q))
        {
            var byQuery = mem.FindBestMatchByNameOrSpan(q);
            if (byQuery != null) return byQuery;

            var byQueryCat = mem.GetMostRecentByCategory(q);
            if (byQueryCat != null) return byQueryCat;
        }

        return null;
    }
}
